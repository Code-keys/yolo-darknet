// IniFile.cpp   
#include "IniFile.h"   
  
void CIniFile::Init()  
{  
    m_unMaxSection = 512;  
    m_unSectionNameMaxSize = 33; // 32位UID串   
}  
  
CIniFile::CIniFile()  
{  
    Init();  
}  
  
CIniFile::CIniFile(LPCTSTR szFileName)  
{  
    // (1) 绝对路径，需检验路径是否存在   
    // (2) 以"./"开头，则需检验后续路径是否存在   
    // (3) 以"../"开头，则涉及相对路径的解析   
      
    Init();  
  
    // 相对路径   
    m_szFileName = szFileName;  
}  
  
CIniFile::~CIniFile()    
{  
      
}  
  
void CIniFile::SetFileName(LPCTSTR szFileName)  
{  
    m_szFileName = szFileName;  
}  
  
DWORD CIniFile::GetProfileSectionNames(vector<wstring> &strArray)  
{  
    int nAllSectionNamesMaxSize = m_unMaxSection*m_unSectionNameMaxSize+1;  
    wchar_t *pszSectionNames = new wchar_t[nAllSectionNamesMaxSize];  
    DWORD dwCopied = 0;  
    dwCopied = ::GetPrivateProfileSectionNames(pszSectionNames, nAllSectionNamesMaxSize, m_szFileName.c_str());  
    strArray.clear();  
    wchar_t *pSection = pszSectionNames;  
    do   
    {  
        wstring  szSection = pSection;
        if (szSection.length() < 1)  
        {  
            delete[] pszSectionNames;  
            return dwCopied;  
        }  
        strArray.push_back(szSection);  
  
        pSection = pSection + szSection.length() + 1; // next section name   
    } while (pSection && pSection<pszSectionNames+nAllSectionNamesMaxSize);  
  
    delete[] pszSectionNames;  
    return dwCopied;  
}  
  
DWORD CIniFile::GetProfileString(LPCTSTR lpszSectionName, LPCTSTR lpszKeyName, wstring& szKeyValue)  
{  
    DWORD dwCopied = 0;  
	wchar_t* keyvalue = new wchar_t[MAX_PATH];
    dwCopied = ::GetPrivateProfileString(lpszSectionName, lpszKeyName, L"",   
        keyvalue, MAX_PATH, m_szFileName.c_str());  
	szKeyValue = keyvalue;
    delete keyvalue;  
  
    return dwCopied;  
}  
  
int CIniFile::GetProfileInt(LPCTSTR lpszSectionName, LPCTSTR lpszKeyName)  
{  
    int nKeyValue = ::GetPrivateProfileInt(lpszSectionName, lpszKeyName, 0, m_szFileName.c_str());  
      
    return nKeyValue;  
}  

BOOL CIniFile::SetProfileString(LPCTSTR lpszSectionName, LPCTSTR lpszKeyName, LPCTSTR lpszKeyValue)  
{  
    return ::WritePrivateProfileString(lpszSectionName, lpszKeyName, lpszKeyValue, m_szFileName.c_str());  
}  
  
BOOL CIniFile::SetProfileInt(LPCTSTR lpszSectionName, LPCTSTR lpszKeyName, int nKeyValue)  
{  
    wchar_t szKeyValue[256];  
    wsprintf(szKeyValue, L"%d", nKeyValue);  
  
    return ::WritePrivateProfileString(lpszSectionName, lpszKeyName, szKeyValue, m_szFileName.c_str());  
}  
  
BOOL CIniFile::DeleteSection(LPCTSTR lpszSectionName)  
{  
    return ::WritePrivateProfileSection(lpszSectionName, NULL, m_szFileName.c_str());  
}  
  
BOOL CIniFile::DeleteKey(LPCTSTR lpszSectionName, LPCTSTR lpszKeyName)  
{  
    return ::WritePrivateProfileString(lpszSectionName, lpszKeyName, NULL, m_szFileName.c_str());  
}  

//https://blog.csdn.net/lewutian/article/details/6787048